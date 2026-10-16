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
    assert [script.stem for script in SCRIPTS] == ["modify", "plot", "read", "write"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.stem)
def test_an_example_runs_and_says_something(
    script: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if script.stem == "plot":
        pytest.importorskip("matplotlib")
        monkeypatch.setenv("MPLBACKEND", "Agg")
    # write.py takes the folder to write to, and plot.py the file of the figure.
    arguments = {"write": [str(tmp_path / "out")], "plot": [str(tmp_path / "figure.png")]}
    monkeypatch.setattr(sys, "argv", [str(script), *arguments.get(script.stem, [])])

    runpy.run_path(str(script), run_name="__main__")

    assert capsys.readouterr().out.strip()
