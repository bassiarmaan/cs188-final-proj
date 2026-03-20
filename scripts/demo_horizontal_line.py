#!/usr/bin/env python3
"""
Horizontal row assembly (full rollout, not only env smoke tests). Requires OSC_POSE.

  python scripts/demo_horizontal_line.py
  python scripts/demo_horizontal_line.py --render
"""

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
from line_assembly.horizontal_line_policy import HorizontalLineAssemblyPolicy  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cubes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="After the arm parks up, keep the window open this long (only with --render)",
    )
    args = parser.parse_args()

    # Joint-velocity control works poorly for this task; use OSC_POSE (same as CA1).
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
        # Should match HorizontalLineAssemblyPolicy.inter_cube_gap so eval and policy agree.
        horizontal_line_eval_gap=0.008,
    )

    obs = env.reset()
    policy = HorizontalLineAssemblyPolicy(env)

    steps = 0
    while steps < args.max_steps and not policy.finished:
        action = policy.get_action(obs)
        obs, _reward, done, info = env.step(action)
        if args.render:
            env.render()
        steps += 1
        if done:
            break

    ev = env.evaluate_horizontal_line()
    print(f"steps={steps} finished_policy={policy.finished} horizontal_line_ok={ev['ok']}")
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
