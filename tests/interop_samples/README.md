# Interoperability Samples

`input_iso2_ascii.sdf` was written by sdfio. The `mm11_*` files were
produced by importing it into Digital Surf MountainsMap 11 and re-exporting
as each dialect/format/data-type combination. `mm11_diagram.png` is
MountainsMap's own rendering of the input file.

Known quirks in the `mm11_*` files (not sdfio bugs):

- The trailer is empty in every file; this exporter doesn't write one.
- Standard non-compliance: the binary files' `ManufacID` field is not
  space-padded past the valid string content ("the string shall be filled
  with spaces"); the unused bytes are whatever was in memory. sdfio reads
  this leniently, truncating at the first NUL byte.
- `binary64` values differ between the ASCII and binary export of the same
  data at the ~1e-19 absolute (~1e-15 relative) level -- the ASCII format's
  own known precision limit (15 mandated significant digits, one short of
  the 17 an IEEE 754 double needs to round-trip exactly), not an export bug.
