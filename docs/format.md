<!--
SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>

SPDX-License-Identifier: MIT
-->

# SDF Format

A surface data file (SDF) stores the heights of a profile or of an areal surface measurement on a
rectangular grid with equal steps along x and y. It is used as a software measurement standard,
also called a softgauge: reference data with which the software of surface texture measuring
instruments is verified. It is also used as an interchange format for topography data.

The format was proposed in 1993 as the BCR format, in the report of the European project that
developed methods for the characterisation of roughness in three dimensions (EUR 15178 EN), and is
standardized in ISO 25178-71. The file extension is `.sdf`. A file is a header, the data area and
a trailer, in an ASCII or a binary representation.

## Revisions and Versions

The first characters of a file, the magic, say which representation it has and which version of
which document it follows. The first character is `a` for the ASCII representation and `b` for the
binary one. It is followed by the document and the version, as in `aISO-2.0` or `bBCR-1.0`.
sdfio reads and writes three revisions:

| Revision | Defined by | Differences |
| --- | --- | --- |
| `BCR-1.0` | The 1993 proposal | Unsigned integer types. The maximum of a type marks a point that is not measured. ASCII files may have comments, and may name the data type instead of giving its code. Times have no time zone. |
| `ISO-1.0` | ISO 25178-71:2017 and ISO 25178-71:2026 | The data types int16, int32 and binary64. The minimum of a type marks a point that is not measured, and the ASCII files write `BAD` for it. Times have no time zone. |
| `ISO-2.0` | ISO 25178-71:2026 | The numbers of columns and rows are 4 bytes instead of 2, so the binary files hold up to 4294967295 of them instead of 65535 (`Revision.max_binary_count`); the standard sets no limit for the ASCII files. The data types int8 and binary32 are added. Times are UTC, and the trailer, if there is one, has the `Name = Value` form of the header. |

The header has fields for a compression and a checksum of the data. ISO-1.0 and ISO-2.0 have only the
value for none, and BCR-1.0 also defines run-length-limited compression and a checksum of each
profile. sdfio does not support them: it reads and writes only files without compression and
without checksums, in all revisions.

## Records

The three records are:

| Record | Content |
| --- | --- |
| Header | The version number and the fields in the next section. |
| Data area | The coded heights, row by row. |
| Trailer | Free text that belongs to the measurement, such as the name of the operator. It is optional and has no length of its own: it extends to the end of the file. |

In a binary file the header has a fixed length, 81 bytes for the first versions and 85 bytes for
ISO-2.0, and nothing separates the records: the header says how long the data area is, and what
follows it is the trailer. In an ASCII file each record ends with a line that has only the
character `*`, and the last of these lines ends the file.

## Header Fields

| Field | Content |
| --- | --- |
| `ManufacID` | Names where the data comes from, in up to 10 characters. |
| `CreateDate` | When the data was generated, as `DDMMYYYYhhmm`. All zeros stand for a time that is not recorded. sdfio holds the time as UTC in memory, and writes it as UTC for `ISO-2.0` and as local time for the older versions. |
| `ModDate` | When the file was last modified, in the same form. |
| `NumPoints` | The number of columns of the matrix. |
| `NumProfiles` | The number of rows of the matrix. It is 1 for a profile, which `is_profile` of the file tests. |
| `Xscale` | The sampling interval along x in metres. |
| `Yscale` | The sampling interval along y in metres. It has no meaning for a profile. |
| `Zscale` | The factor that turns a coded height into metres. It is positive. |
| `Zresolution` | The quantization step of the instrument in metres, or a negative number if it is not known. |
| `Compression` | The type of compression: 0 for none, and in BCR-1.0 also 1 for run-length-limited. |
| `DataType` | The code of the data type of the data area. |
| `CheckType` | The type of checksum: 0 for none, and in BCR-1.0 also 1 for a checksum of each profile. |

The names `NumPoints` and `NumProfiles` come from the first versions, when the format was meant for
tactile instruments that measure profiles. A profile has `NumPoints` points, and a surface has
`NumProfiles` profiles of that many points. The binary header has the fields in this order, with
the version number first, as 8 characters, then `ManufacID` as 10, and the two dates as 12 each. The
numbers are little endian: `NumPoints` and `NumProfiles` are unsigned integers of 2 bytes, or 4
bytes in ISO-2.0, the scales and the resolution are binary64, and the last three fields are one
byte each.

An ASCII header is a list of lines `Name = value`, after the line with the version number, in any
order, and the names do not depend on the case. A date is the 12 digits of the field, and a number is
written in scientific notation.

## Data Area and Coordinates

The data area is the matrix of the coded heights with the rows after each other. A coded height
`w` at the row `i` and the column `j`, counted from 1, is at the position

```text
x = (j - 1) * Xscale     y = (i - 1) * Yscale     z = w * Zscale
```

in a right-handed coordinate system: `x` grows with the column and `y` with the row. In a file
read by sdfio, `data` is the array of the heights in metres and `x_axis` and `y_axis` have the
coordinates. `points()` of the file combines them in an array of shape `(rows, columns, 3)`. The
first row has `y = 0` in all revisions. The proposal of 1993 did not say in which direction the
profiles are stored. sdfio follows MountainsMap, which shows the first profile of a BCR file at the
maximum `y`, and reverses the rows when it reads and writes that revision.

## Data Types

The data type says how a coded height is stored. In an ASCII file integers are written in decimal,
binary32 with 7 significant digits and binary64 with 15.

| Code | Type | Bytes | BCR-1.0 | ISO-1.0 | ISO-2.0 |
| --- | --- | --- | --- | --- | --- |
| 0 | uint8 | 1 | yes | no | no |
| 1 | uint16 | 2 | yes | no | no |
| 2 | uint32 | 4 | yes | no | no |
| 3 | binary32 | 4 | yes | no | yes |
| 4 | int8 | 1 | yes | no | yes |
| 5 | int16 | 2 | yes | yes | yes |
| 6 | int32 | 4 | yes | yes | yes |
| 7 | binary64 | 8 | yes | yes | yes |

The heights of an integer type are coded as `z / Zscale`, rounded, so `Zscale` trades the range
of the type against the resolution. A floating point type can store metres with `Zscale` 1, or any
other scale. sdfio chooses the smallest scale that fits the data when it is asked to.

## Points That Are Not Measured

A point that was not measured, or is spurious, has a value in the data area that no valid height
can have. For ISO that is the minimum value of the data type, and the ASCII representation writes
`BAD`. The BCR proposal used the maximum value of the type instead, and ASCII files of BCR have
that value as a number. sdfio reads such a point as `NaN` and writes `NaN` as that value, so a
valid height cannot be the value that is reserved.

## Trailer

The trailer is text in 7-bit ASCII. Its content is not defined in the first versions. ISO-2.0
requires the lines of the trailer to have the same form as those of the header, `Name = Value`,
and sdfio offers them as `trailer_fields`.

## Line Ends and Case

ASCII files of ISO end each line with CR and LF. Files of the BCR revision may end lines with CR
or LF alone, and sdfio reads them. The magic of a BCR file may have an upper case first
character, which sdfio accepts for all revisions.
