<!--
SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>

SPDX-License-Identifier: MIT
-->

# Examples

The scripts are in the `examples/` folder. Short examples of the common use cases are also at the
top of the API reference.

## Writing

Surfaces and profiles, with metadata and a trailer, in different revisions, data types and
representations.

```{literalinclude} ../examples/write.py
:language: python
```

## Reading

The heights, the points, the header and the trailer of a file, from a path, a stream or bytes.

```{literalinclude} ../examples/read.py
:language: python
```

## Modifying

Changing the values, making files of parts of a file, and converting it to another revision, data
type or representation.

```{literalinclude} ../examples/modify.py
:language: python
```

## Plotting

```{literalinclude} ../examples/plot.py
:language: python
```
