#!/usr/bin/env python3
"""Smiley-face layout (nine cubes, flat on the table). Targets come from layout_patterns; policy is GenericAssemblyPolicy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import line_assembly  # noqa: E402, F401
from line_assembly.color_presets import cycle_for_preset  # noqa: E402

import play_task_menu as ptm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-steps", type=int, default=80000)
    ap.add_argument("--hold-seconds", type=float, default=3.0)
    ap.add_argument("--pick", choices=("color", "x"), default="color")
    ap.add_argument(
        "--color",
        type=str,
        default="default",
        help="Cube colors: default, blue, red, green, yellow, magenta",
    )
    args = ap.parse_args()

    rgba = cycle_for_preset(args.color)
    n = ptm._cube_count_for_task("smiley")
    env = ptm._make_env(
        cube_count=n,
        rgba_cycle=rgba,
        horizon=args.max_steps + 50,
        render=args.render,
        placement_profile="calibration",
    )
    obs = env.reset()
    policy = ptm._build_policy("smiley", env, args.pick)
    print("Smiley face (9 cubes, flat layout)")
    ptm._run_policy(
        env,
        policy,
        obs=obs,
        max_steps=args.max_steps,
        render=args.render,
        hold_seconds=args.hold_seconds,
    )
    env.close()


if __name__ == "__main__":
    main()
