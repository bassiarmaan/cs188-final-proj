"""
Wire text → task label → fake "plan" → which policy to run.

Nothing fancy: regex-ish keyword matching for now. If you drop in an LLM later, keep the
same datatypes so tests don't explode. Good enough for the writeup pipeline too
(parse → plan → control).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """What we think the user meant."""

    STACK_VERTICAL = "stack_vertical"
    LINE_HORIZONTAL = "line_horizontal"
    SORT_BY_COLOR = "sort_by_color"
    SORT_BY_SIZE = "sort_by_size"


@dataclass
class ParsedInstruction:
    raw_text: str
    task_type: TaskType
    """Parsed count from \"three blocks\" etc., or None."""
    suggested_cube_count: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Subgoal:
    """Buzzword for the report / logging; often one policy run covers all of it."""

    name: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlBinding:
    """Which policy class + env kwargs."""

    policy_id: str
    policy_kwargs: dict[str, Any]
    env_overrides: dict[str, Any]
    sim_supported: bool = True


def _extract_small_int(text: str) -> int | None:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    lower = text.lower()
    for w, n in words.items():
        if re.search(rf"\b{w}\b", lower):
            return n
    m = re.search(r"\b(\d{1,2})\b", lower)
    if m:
        return int(m.group(1))
    return None


def parse_instruction(text: str) -> ParsedInstruction:
    # Order matters: size beats color beats stack vs line messiness.
    t = text.strip()
    lower = t.lower()
    notes: list[str] = []
    n_cubes = _extract_small_int(lower)

    def _w(word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", lower) is not None

    size_hit = any(
        k in lower
        for k in (
            "sort by size",
            "order by size",
            "biggest",
            "smallest",
            "largest",
            "by size",
        )
    )
    color_hit = any(
        k in lower
        for k in (
            "sort by color",
            "order by color",
            "rainbow",
            "chromatic",
            "hue",
            "color order",
        )
    ) or (
        _w("color")
        and any(_w(w) for w in ("sort", "order", "arrange", "organize"))
    )
    stack_hit = any(
        k in lower for k in ("stack", "tower", "pile", "column", "vertically", "on top of each")
    )
    line_hit = any(
        k in lower
        for k in (
            "in a row",
            "in a line",
            "line up",
            "line them",
            "horizontal",
            "side by side",
            "straight line",
        )
    )

    if size_hit:
        return ParsedInstruction(t, TaskType.SORT_BY_SIZE, n_cubes, notes)

    if color_hit:
        if n_cubes is None:
            notes.append("no_count_mentioned_default_env")
        return ParsedInstruction(t, TaskType.SORT_BY_COLOR, n_cubes, notes)

    if stack_hit and not line_hit:
        if n_cubes is None:
            notes.append("no_count_mentioned_default_env")
        return ParsedInstruction(t, TaskType.STACK_VERTICAL, n_cubes, notes)

    if line_hit or re.search(r"\b(row|align)\b", lower):
        if n_cubes is None:
            notes.append("no_count_mentioned_default_env")
        return ParsedInstruction(t, TaskType.LINE_HORIZONTAL, n_cubes, notes)

    raise ValueError(f"no idea what that means: {text!r}")


def build_subgoal_plan(parsed: ParsedInstruction) -> list[Subgoal]:
    # Mostly for screenshots / prose in the doc.
    if parsed.task_type is TaskType.STACK_VERTICAL:
        return [
            Subgoal(
                "perceive_cube_poses",
                "Read cube poses from simulation state.",
                {},
            ),
            Subgoal(
                "stack_by_vertical_slots",
                "Pick each block and place at shared (x,y) with increasing z.",
                {"axis": "z", "inter_cube_gap_m": 0.008},
            ),
            Subgoal(
                "verify_vertical_stack",
                "Check center spacing along +z vs nominal pitch.",
                {},
            ),
        ]

    if parsed.task_type is TaskType.LINE_HORIZONTAL:
        return [
            Subgoal("perceive_cube_poses", "Read cube poses from simulation state.", {}),
            Subgoal(
                "order_picks_spatial",
                "Pick cubes in west-to-east (x) order for a left-to-right row.",
                {"pick_sort": "x"},
            ),
            Subgoal(
                "place_adjacent_along_x",
                "Place cubes in adjacent slots along +x with optional gap.",
                {"axis": "x", "inter_cube_gap_m": 0.008},
            ),
            Subgoal("verify_horizontal_line", "Run geometric adjacency check along x.", {}),
        ]

    if parsed.task_type is TaskType.SORT_BY_COLOR:
        return [
            Subgoal("perceive_cube_poses", "Read cube poses from simulation state.", {}),
            Subgoal(
                "order_picks_by_palette_index",
                "Pick cubes ordered by env palette index (color cycle), tie-break by x.",
                {"pick_sort": "color"},
            ),
            Subgoal(
                "place_adjacent_along_x",
                "Place cubes in a row; final order encodes color sorting.",
                {"axis": "x", "inter_cube_gap_m": 0.008},
            ),
            Subgoal("verify_horizontal_line", "Same geometry as a row task.", {}),
        ]

    if parsed.task_type is TaskType.SORT_BY_SIZE:
        return [
            Subgoal("perceive_cube_extents", "Estimate or read each block's half-extents.", {}),
            Subgoal(
                "order_picks_by_extent",
                "Order picks descending by size (largest first) or user preference.",
                {"key": "max_half_extent"},
            ),
            Subgoal(
                "place_per_plan",
                "Place blocks at goal layout (requires variable-size objects in env).",
                {"status": "planned_only"},
            ),
        ]

    raise RuntimeError(f"Unhandled task type: {parsed.task_type}")


def control_binding_for_plan(parsed: ParsedInstruction) -> ControlBinding:
    # Actually hook up to HorizontalLine / VerticalLine policies.
    gap = 0.008
    base_env = {
        "horizontal_line_eval_gap": gap,
        "vertical_stack_eval_gap": gap,
    }
    if parsed.suggested_cube_count is not None:
        base_env["cube_count"] = max(1, min(8, int(parsed.suggested_cube_count)))

    if parsed.task_type is TaskType.STACK_VERTICAL:
        return ControlBinding(
            policy_id="vertical_line_assembly",
            policy_kwargs={"inter_cube_gap": gap, "pick_sort_mode": "color"},
            env_overrides={**base_env, "placement_profile": "calibration"},
        )

    if parsed.task_type is TaskType.LINE_HORIZONTAL:
        return ControlBinding(
            policy_id="horizontal_line_assembly",
            policy_kwargs={"inter_cube_gap": gap, "pick_sort_mode": "x"},
            env_overrides={**base_env, "placement_profile": "calibration"},
        )

    if parsed.task_type is TaskType.SORT_BY_COLOR:
        return ControlBinding(
            policy_id="horizontal_line_assembly",
            policy_kwargs={"inter_cube_gap": gap, "pick_sort_mode": "color"},
            env_overrides={**base_env, "placement_profile": "calibration"},
        )

    if parsed.task_type is TaskType.SORT_BY_SIZE:
        return ControlBinding(
            policy_id="none",
            policy_kwargs={},
            env_overrides=base_env,
            sim_supported=False,
        )

    raise RuntimeError(f"Unhandled task type: {parsed.task_type}")


def instantiate_policy(env: Any, binding: ControlBinding):
    # Needs OSC_POSE like the rest of the demos.
    from .horizontal_line_policy import HorizontalLineAssemblyPolicy
    from .vertical_line_policy import VerticalLineAssemblyPolicy

    if not binding.sim_supported:
        raise NotImplementedError("size sort needs different block sizes in the env first")
    pid = binding.policy_id
    kw = dict(binding.policy_kwargs)
    if pid == "vertical_line_assembly":
        return VerticalLineAssemblyPolicy(env, **kw)
    if pid == "horizontal_line_assembly":
        return HorizontalLineAssemblyPolicy(env, **kw)
    raise ValueError(f"Unknown policy_id: {pid}")


def run_parsed_instruction(
    env: Any,
    parsed: ParsedInstruction,
    *,
    max_steps: int,
    render: bool = False,
) -> tuple[int, dict[str, Any]]:
    # Convenience wrapper for scripts; returns (step count, little summary dict).
    binding = control_binding_for_plan(parsed)
    policy = instantiate_policy(env, binding)
    obs = env.reset()
    steps = 0
    info: dict[str, Any] = {}
    while steps < max_steps and not policy.finished:
        obs, _, done, info = env.step(policy.get_action(obs))
        if render:
            env.render()
        steps += 1
        if done:
            break
    summary: dict[str, Any] = {
        "steps": steps,
        "finished_policy": policy.finished,
        "task_type": parsed.task_type.value,
    }
    if parsed.task_type is TaskType.STACK_VERTICAL:
        summary["evaluate"] = env.evaluate_vertical_stack()
    elif parsed.task_type in (TaskType.LINE_HORIZONTAL, TaskType.SORT_BY_COLOR):
        summary["evaluate"] = env.evaluate_horizontal_line()
    summary["info_tail"] = info
    return steps, summary
