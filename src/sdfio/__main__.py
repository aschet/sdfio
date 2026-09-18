# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Allow running the CLI as ``python -m sdfio``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
