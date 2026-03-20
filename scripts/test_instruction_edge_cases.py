#!/usr/bin/env python3
"""Weird strings, precedence, should-raise cases for parse_instruction."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_assembly.instruction_bridge import TaskType, parse_instruction  # noqa: E402


def main():
    # Extra whitespace / casing
    p = parse_instruction("  STACK the Blocks  ")
    assert p.task_type is TaskType.STACK_VERTICAL

    # Digit count
    p = parse_instruction("use 2 cubes in a row")
    assert p.task_type is TaskType.LINE_HORIZONTAL
    assert p.suggested_cube_count == 2

    # Size wins over color when both mentioned
    p = parse_instruction("sort by color and by size")
    assert p.task_type is TaskType.SORT_BY_SIZE

    p = parse_instruction("vertical")
    assert p.task_type is TaskType.STACK_VERTICAL

    # Gibberish
    for bad in ("", "hello", "do the thing"):
        try:
            parse_instruction(bad)
            raise AssertionError(f"expected failure for {bad!r}")
        except ValueError:
            pass

    print("instruction edge cases: OK")


if __name__ == "__main__":
    main()
