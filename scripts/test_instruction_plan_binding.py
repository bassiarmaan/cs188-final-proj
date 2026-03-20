#!/usr/bin/env python3
"""Plan + binding wiring without spinning up MuJoCo."""

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
    parse_instruction,
)


def _test_plan_shapes():
    for text, expected_names_tail in [
        ("stack the blocks", ("verify_vertical_stack",)),
        ("line them up in a row", ("verify_horizontal_line",)),
        ("sort by color", ("verify_horizontal_line",)),
        ("sort by size", ("place_per_plan",)),
    ]:
        parsed = parse_instruction(text)
        plan = build_subgoal_plan(parsed)
        assert len(plan) >= 2, text
        assert plan[0].name.startswith("perceive"), text
        last = plan[-1].name
        assert last == expected_names_tail[0], (text, last, expected_names_tail)


def _test_bindings_distinct_policies():
    stack = control_binding_for_plan(parse_instruction("stack"))
    line = control_binding_for_plan(parse_instruction("in a row"))
    color = control_binding_for_plan(parse_instruction("sort by color"))
    size = control_binding_for_plan(parse_instruction("sort by size"))

    assert stack.policy_id == "vertical_line_assembly"
    assert stack.sim_supported
    assert line.policy_id == "horizontal_line_assembly"
    assert line.policy_kwargs["pick_sort_mode"] == "x"
    assert color.policy_id == "horizontal_line_assembly"
    assert color.policy_kwargs["pick_sort_mode"] == "color"
    assert size.sim_supported is False
    assert size.policy_id == "none"


def _test_cube_count_override():
    p = parse_instruction("stack five blocks")
    b = control_binding_for_plan(p)
    assert b.env_overrides.get("cube_count") == 5


def _test_instantiate_rejects_size():
    import line_assembly  # noqa: F401
    from line_assembly.instruction_bridge import instantiate_policy

    binding = control_binding_for_plan(parse_instruction("order by size"))
    try:
        instantiate_policy(None, binding)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def main():
    _test_plan_shapes()
    _test_bindings_distinct_policies()
    _test_cube_count_override()
    _test_instantiate_rejects_size()
    print("plan + binding: OK")


if __name__ == "__main__":
    main()
