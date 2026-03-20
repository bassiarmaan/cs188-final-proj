#!/usr/bin/env python3
"""Vertical stack of cubes in one column. Controller setup matches demo_horizontal_line (OSC_POSE)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import robosuite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402

import line_assembly  # noqa: E402, F401
from line_assembly.vertical_line_policy import VerticalLineAssemblyPolicy  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cubes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="Keep viewer open after parking (only with --render)",
    )
    args = parser.parse_args()

    ctrl = load_controller_config(default_controller="OSC_POSE")

    env = robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        use_camera_obs=False,
        horizon=max(args.max_steps, 1),
        control_freq=20,
        cube_count=args.cubes,
        placement_profile="calibration",
        horizontal_reward_mode="none",
        vertical_stack_eval_gap=0.008,
    )

    obs = env.reset()
    policy = VerticalLineAssemblyPolicy(env)

    steps = 0
    while steps < args.max_steps and not policy.finished:
        action = policy.get_action(obs)
        obs, _reward, done, _info = env.step(action)
        if args.render:
            env.render()
        steps += 1
        if done:
            break

    ev = env.evaluate_vertical_stack()
    print(
        f"steps={steps} finished_policy={policy.finished} vertical_stack_ok={ev['ok']}"
    )
    if not ev["ok"]:
        print("detail:", ev)

    if args.render and args.hold_seconds > 0:
        n_hold = int(args.hold_seconds * env.control_freq)
        hold_action = np.zeros(env.action_dim)
        hold_action[6:] = -1.0
        for _ in range(n_hold):
            obs, _, _, _ = env.step(hold_action)
            env.render()

    env.close()


if __name__ == "__main__":
    main()
