# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The scripts in ``examples/`` run, so that what the documentation shows keeps working."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))


def test_there_is_an_example_for_each_topic() -> None:
    assert [script.stem for script in SCRIPTS] == [
        "01_surfaces",
        "02_profiles",
        "03_missing_points",
        "04_revisions_and_data_types",
        "05_header_and_trailer",
        "06_reading_files",
        "07_modifying_values",
        "08_plotting",
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.stem)
def test_an_example_runs_and_says_something(
    script: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", [str(script)])

    runpy.run_path(str(script), run_name="__main__")

    assert capsys.readouterr().out.strip()
