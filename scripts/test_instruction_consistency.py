#!/usr/bin/env python3
"""Paranoid pass: every TaskType has something to run (or explicitly can't)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from line_assembly.instruction_bridge import (  # noqa: E402
    TaskType,
    build_subgoal_plan,
    control_binding_for_plan,
    instantiate_policy,
)


def _canonical_phrase(task: TaskType) -> str:
    return {
        TaskType.STACK_VERTICAL: "stack the blocks",
        TaskType.LINE_HORIZONTAL: "line up in a row",
        TaskType.SORT_BY_COLOR: "sort by color",
        TaskType.SORT_BY_SIZE: "sort by size",
    }[task]


def main():
    from line_assembly.instruction_bridge import parse_instruction

    for task in TaskType:
        text = _canonical_phrase(task)
        parsed = parse_instruction(text)
        assert parsed.task_type is task
        plan = build_subgoal_plan(parsed)
        assert all(sg.name and sg.description for sg in plan)
        binding = control_binding_for_plan(parsed)
        assert binding.policy_id or not binding.sim_supported
        if binding.sim_supported:
            # Policy import path smoke (env None would fail at runtime — only import check)
            from line_assembly.horizontal_line_policy import HorizontalLineAssemblyPolicy
            from line_assembly.vertical_line_policy import VerticalLineAssemblyPolicy

            assert binding.policy_id in (
                "horizontal_line_assembly",
                "vertical_line_assembly",
            )
            _ = HorizontalLineAssemblyPolicy
            _ = VerticalLineAssemblyPolicy
        else:
            try:
                instantiate_policy(None, binding)
            except NotImplementedError:
                pass
            else:
                raise AssertionError("size task should not instantiate")

    print("instruction consistency: OK (all TaskTypes wired)")


if __name__ == "__main__":
    main()
