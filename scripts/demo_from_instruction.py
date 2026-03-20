#!/usr/bin/env python3
"""Parse a single instruction string; optionally print the JSON plan, then run it through instruction_bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import robosuite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402

import line_assembly  # noqa: E402, F401
from line_assembly.instruction_bridge import (  # noqa: E402
    build_subgoal_plan,
    control_binding_for_plan,
    parse_instruction,
    run_parsed_instruction,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-steps", type=int, default=25000)
    ap.add_argument("--print-json", action="store_true", help="Dump plan + binding to stdout.")
    args = ap.parse_args()

    parsed = parse_instruction(args.text)
    plan = build_subgoal_plan(parsed)
    binding = control_binding_for_plan(parsed)

    if args.print_json:
        print(
            json.dumps(
                {
                    "task_type": parsed.task_type.value,
                    "suggested_cube_count": parsed.suggested_cube_count,
                    "subgoals": [sg.name for sg in plan],
                    "policy_id": binding.policy_id,
                    "sim_supported": binding.sim_supported,
                },
                indent=2,
            )
        )

    if not binding.sim_supported:
        print("Task is planned but not executable in sim yet:", parsed.task_type.value)
        sys.exit(2)

    ctrl = load_controller_config(default_controller="OSC_POSE")
    cube_count = binding.env_overrides.get("cube_count", 3)
    env = robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        use_camera_obs=False,
        horizon=max(args.max_steps, 1) + 10,
        control_freq=20,
        cube_count=cube_count,
        placement_profile="calibration",
        horizontal_reward_mode="none",
        horizontal_line_eval_gap=0.008,
        vertical_stack_eval_gap=0.008,
    )
    for k, v in binding.env_overrides.items():
        if hasattr(env, k):
            setattr(env, k, v)

    steps, summary = run_parsed_instruction(
        env, parsed, max_steps=args.max_steps, render=args.render
    )
    print("steps", steps, "summary", {k: summary[k] for k in summary if k != "info_tail"})
    env.close()


if __name__ == "__main__":
    main()
