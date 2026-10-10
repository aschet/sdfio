# Sample Files

Synthetic SDF files for each dialect of the format, in the folder that is named after the document
that defines the dialect. ISO 25178-71:2026 specifies the versions 1.0 and 2.0, so they share a
folder. The name of a file starts with the dialect:

| Folder | Dialect | Start of the file name |
| --- | --- | --- |
| `iso25178-71-2026/` | ISO-1.0 | `iso1_` |
| `iso25178-71-2026/` | ISO-2.0 | `iso2_` |
| `bcr-1.0/` | BCR-1.0 | `bcr1_` |

Every file exists in the ASCII representation (`_ascii`) and in the binary one (`_binary`), as in
`iso2_surface_binary.sdf`. The surfaces have 48 columns and 32 rows with the sampling intervals
2 µm along x and 3 µm along y, and the profiles have 128 points. Missing points are the rows 10 to
13 and the columns 20 to 29 of a surface, and the points 60 to 63 of a profile.

| Name | Data | Dialects |
| --- | --- | --- |
| `profile` | A profile with missing points, binary64 | all |
| `surface` | A surface with a region of missing points and a trailer, int16 | all |
| `surface_int32` | The same surface, int32 | all |
| `surface_binary64` | The same surface, binary64 | all |
| `surface_binary32`, `surface_int8` | The same surface, in the data types that version 2.0 adds | ISO-2.0 |
| `surface_uint16` | The same surface above zero, in an unsigned type | BCR-1.0 |

The trailer of `surface` is in the tagged `Name = Value` format in all dialects. The creation and
modification date is 2026-01-01 12:00, in UTC for ISO-2.0 and without a time zone for the others,
so that the files do not depend on where they are made.

`generate.py` makes the files.
