#!/usr/bin/env python3
"""Random actions, just proves the env loads. Won't stack anything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import robosuite  # noqa: E402

import line_assembly  # noqa: E402, F401  # side effect: register env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Use on-screen renderer")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--cubes", type=int, default=6)
    parser.add_argument(
        "--placement",
        default="calibration",
        choices=("calibration", "lift_default", "table_uniform", "horizontal_line", "vertical_stack"),
        help="Spawn profile: default matches 2_calibration_starter bounds on a Lift-sized table",
    )
    args = parser.parse_args()

    env = robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        use_camera_obs=False,
        horizon=max(args.steps, 1),
        control_freq=20,
        cube_count=args.cubes,
        placement_profile=args.placement,
    )

    env.reset()
    low, high = env.action_spec
    for _ in range(args.steps):
        action = np.random.uniform(low, high)
        env.step(action)
        if args.render:
            env.render()

    env.close()
    print("smoke_test_env: OK")


if __name__ == "__main__":
    main()
