#!/usr/bin/env python3
"""Custom design from text prompt: GPT returns a 5x7 bitmap (max 12 cubes), robot arranges cubes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpt_api  # noqa: E402
import line_assembly  # noqa: E402, F401
from line_assembly.color_presets import cycle_for_preset  # noqa: E402
from line_assembly.layout_patterns import (  # noqa: E402
    set_smiley_rows,
    smiley_cube_count,
)

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
    ap.add_argument(
        "prompt",
        nargs="?",
        help="Design description (or omit to be prompted interactively)",
    )
    args = ap.parse_args()

    prompt = (args.prompt or input("Describe the design (e.g. 'heart', 'smiley face'): ")).strip()
    if not prompt:
        print("No prompt entered. Exiting.")
        sys.exit(1)

    print("Fetching bitmap from GPT...")
    try:
        bitmap = gpt_api.fetch_bitmap(prompt, nrows=5, ncols=7)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    set_smiley_rows(bitmap)
    n_cubes = smiley_cube_count()
    if n_cubes > 12:
        print(f"Error: bitmap has {n_cubes} cubes (max 12). Try a simpler design.")
        sys.exit(1)
    print(f"Bitmap ({n_cubes} cubes):")
    for row in bitmap:
        print(f"  {row}")

    rgba = cycle_for_preset(args.color)
    # Use table_uniform for 9+ cubes; calibration region is too small
    profile = "table_uniform" if n_cubes > 8 else "calibration"
    env = ptm._make_env(
        cube_count=n_cubes,
        rgba_cycle=rgba,
        horizon=args.max_steps + 50,
        render=args.render,
        placement_profile=profile,
    )
    obs = env.reset()
    # set_smiley_rows already called; _build_policy("smiley") uses smiley_world_slots → our bitmap
    policy = ptm._build_policy("smiley", env, args.pick)

    print(f"Custom design: {n_cubes} cubes")
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
