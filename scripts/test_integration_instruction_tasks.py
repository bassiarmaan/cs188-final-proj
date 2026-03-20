#!/usr/bin/env python3
"""End-to-end NL → policy → sim. Slow-ish; caps steps per phrase."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import robosuite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402

import line_assembly  # noqa: E402, F401
from line_assembly.instruction_bridge import (  # noqa: E402
    parse_instruction,
    run_parsed_instruction,
)


def _make_env(cube_count: int = 3):
    ctrl = load_controller_config(default_controller="OSC_POSE")
    return robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        horizon=40000,
        control_freq=20,
        cube_count=cube_count,
        placement_profile="calibration",
        horizontal_reward_mode="none",
        horizontal_line_eval_gap=0.008,
        vertical_stack_eval_gap=0.008,
    )


def _apply_binding_env_overrides(env, binding):
    for k, v in binding.env_overrides.items():
        if hasattr(env, k):
            setattr(env, k, v)


def main():
    ctrl = load_controller_config(default_controller="OSC_POSE")
    scenarios = [
        # Two cubes: more reliable single-column stack under physics variance.
        ("stack two blocks", "vertical", lambda ev: ev["ok"]),
        ("put four cubes in a row", "horizontal", lambda ev: ev["ok"]),
        ("sort by color", "color_row", lambda ev: ev["ok"]),
    ]
    max_steps = 20000

    for phrase, label, ok_fn in scenarios:
        parsed = parse_instruction(phrase)
        from line_assembly.instruction_bridge import control_binding_for_plan

        binding = control_binding_for_plan(parsed)
        cube_count = binding.env_overrides.get("cube_count", 3)
        env = robosuite.make(
            "LineLayoutBlocksEnv",
            robots="Panda",
            controller_configs=ctrl,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            horizon=max_steps + 10,
            control_freq=20,
            cube_count=cube_count,
            placement_profile="calibration",
            horizontal_reward_mode="none",
            horizontal_line_eval_gap=0.008,
            vertical_stack_eval_gap=0.008,
        )
        _apply_binding_env_overrides(env, binding)
        steps, summary = run_parsed_instruction(env, parsed, max_steps=max_steps, render=False)
        env.close()
        ev = summary["evaluate"]
        assert ok_fn(ev), (label, phrase, steps, summary, ev)
        assert summary["finished_policy"], (label, steps)
        print(f"integration [{label}]: OK steps={steps}")

    print("\nAll integration instruction tasks passed.")


if __name__ == "__main__":
    main()
