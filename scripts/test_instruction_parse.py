#!/usr/bin/env python3
"""String in, TaskType out. No sim. Add phrases when you feel like it."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_assembly.instruction_bridge import TaskType, parse_instruction  # noqa: E402


def _cases():
    # (phrase, expected_task, optional_expected_count)
    return [
        ("Please stack the blocks", TaskType.STACK_VERTICAL, None),
        ("Build a tall tower", TaskType.STACK_VERTICAL, None),
        ("pile them vertically", TaskType.STACK_VERTICAL, None),
        ("Stack three blocks", TaskType.STACK_VERTICAL, 3),
        ("vertical stack please", TaskType.STACK_VERTICAL, None),
        ("Put one on top of the other", TaskType.STACK_VERTICAL, None),
        ("Place them on top of each other", TaskType.STACK_VERTICAL, None),
        ("Line up the cubes in a row", TaskType.LINE_HORIZONTAL, None),
        ("Put them side by side horizontally", TaskType.LINE_HORIZONTAL, None),
        ("Arrange in a straight line", TaskType.LINE_HORIZONTAL, None),
        ("4 cubes in a line", TaskType.LINE_HORIZONTAL, 4),
        ("Sort the blocks by color", TaskType.SORT_BY_COLOR, None),
        ("Order by color — rainbow style", TaskType.SORT_BY_COLOR, None),
        ("Place cubes in color order", TaskType.SORT_BY_COLOR, None),
        ("Sort by size smallest to largest", TaskType.SORT_BY_SIZE, None),
        ("Order blocks by size", TaskType.SORT_BY_SIZE, None),
        ("Put the biggest block first", TaskType.SORT_BY_SIZE, None),
    ]


def _test_precedence_stack_over_line():
    """Stack cues win over weak horizontal wording ("line up", "straight line", …)."""
    p = parse_instruction("stack blocks in a vertical column")
    assert p.task_type is TaskType.STACK_VERTICAL
    p = parse_instruction("line up the cubes vertically")
    assert p.task_type is TaskType.STACK_VERTICAL
    p = parse_instruction("line them in a column")
    assert p.task_type is TaskType.STACK_VERTICAL
    p = parse_instruction("line up in a row")
    assert p.task_type is TaskType.LINE_HORIZONTAL


def _test_unknown_raises():
    try:
        parse_instruction("make coffee and file taxes")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def main():
    for phrase, task, n in _cases():
        p = parse_instruction(phrase)
        assert p.task_type is task, (phrase, p.task_type, task)
        assert p.suggested_cube_count == n, (phrase, p.suggested_cube_count, n)
    _test_precedence_stack_over_line()
    _test_unknown_raises()
    print(f"instruction parse: OK ({len(list(_cases()))} phrases + precedence + unknown)")


if __name__ == "__main__":
    main()
