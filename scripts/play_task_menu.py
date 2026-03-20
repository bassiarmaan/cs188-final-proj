#!/usr/bin/env python3
"""
Interactive launcher: enter a task and optional color, with or without rendering.

  Tasks: h, hc|sort, v, smiley, g2, g3  (smiley is a smaller task; handy to try before the grids)
  Color names can appear anywhere in the line, e.g.  blue v  or  red smiley
  q or quit exits.

Uses OSC_POSE only, consistent with the other demos.
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
from line_assembly.color_presets import cycle_for_preset  # noqa: E402
from line_assembly.generic_assembly_policy import GenericAssemblyPolicy  # noqa: E402
from line_assembly.horizontal_line_policy import (  # noqa: E402
    CubeJob,
    HorizontalLineAssemblyPolicy,
    pick_indices_in_palette_order,
)
from line_assembly.layout_patterns import (  # noqa: E402
    grid_cube_count,
    grid_stack_world_slots,
    smiley_cube_count,
    smiley_world_slots,
)
from line_assembly.vertical_line_policy import VerticalLineAssemblyPolicy  # noqa: E402

_COLOR_WORDS = frozenset(
    (
        "default",
        "multi",
        "rainbow",
        "blue",
        "all_blue",
        "red",
        "green",
        "yellow",
        "magenta",
        "purple",
    )
)


def _pick_order(env, mode: str) -> np.ndarray:
    positions = env.get_cube_world_positions()
    n = positions.shape[0]
    k = max(1, len(env.rgba_cycle))
    if mode == "color":
        return pick_indices_in_palette_order(n, positions[:, 0], k)
    return np.argsort(positions[:, 0])


def _jobs_from_slots(env, slots: list, pick_mode: str) -> list[CubeJob]:
    order = _pick_order(env, pick_mode)
    assert len(order) == len(slots)
    return [CubeJob(int(order[i]), slots[i].copy()) for i in range(len(slots))]


def _make_env(
    *,
    cube_count: int,
    rgba_cycle,
    horizon: int,
    render: bool,
    placement_profile: str = "calibration",
):
    ctrl = load_controller_config(default_controller="OSC_POSE")
    return robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=render,
        has_offscreen_renderer=not render,
        use_camera_obs=False,
        horizon=horizon,
        control_freq=20,
        cube_count=cube_count,
        rgba_cycle=rgba_cycle,
        placement_profile=placement_profile,
        horizontal_reward_mode="none",
        horizontal_line_eval_gap=0.008,
        vertical_stack_eval_gap=0.008,
    )


def _run_policy(env, policy, *, max_steps: int, render: bool, hold_seconds: float):
    obs = env.reset()
    steps = 0
    while steps < max_steps and not policy.finished:
        obs, _, done, info = env.step(policy.get_action(obs))
        if render:
            env.render()
        steps += 1
        if done:
            break
    print(
        f"  steps={steps} finished={policy.finished}  "
        f"h_line_ok={info.get('horizontal_line_ok')}  "
        f"v_stack_ok={info.get('vertical_stack_ok')}"
    )
    if hold_seconds > 0 and render:
        n_hold = int(hold_seconds * env.control_freq)
        hold = np.zeros(env.action_dim)
        hold[6:] = -1.0
        for _ in range(n_hold):
            env.step(hold)
            env.render()


def _parse_line(line: str) -> tuple[str | None, str]:
    parts = line.strip().lower().split()
    if not parts:
        return None, "default"
    color = "default"
    filtered: list[str] = []
    for p in parts:
        if p in _COLOR_WORDS:
            color = "blue" if p == "all_blue" else p
        else:
            filtered.append(p)
    if not filtered:
        return None, color
    return filtered[0], color


def _build_policy(task: str, env, pick_override: str | None):
    gap = 0.008
    if task in ("h", "row", "horizontal"):
        mode = pick_override or "x"
        return HorizontalLineAssemblyPolicy(env, inter_cube_gap=gap, pick_sort_mode=mode)
    if task in ("hc", "hcolor", "rowc", "sort", "colorsort"):
        return HorizontalLineAssemblyPolicy(env, inter_cube_gap=gap, pick_sort_mode="color")
    if task in ("v", "stack", "vertical"):
        mode = pick_override or "color"
        return VerticalLineAssemblyPolicy(env, inter_cube_gap=gap, pick_sort_mode=mode)
    if task in ("g2", "grid2"):
        n = 2
        h = 2
        slots = grid_stack_world_slots(env, n, n, h)
        pick = pick_override or "color"
        jobs = _jobs_from_slots(env, slots, pick)
        return GenericAssemblyPolicy(
            env,
            jobs,
            inter_cube_gap=gap,
            ascend_above_each_slot=True,
            ascend_clear_above_placed_z=0.38,
            hover_slot_z_extra=0.16,
            transit_min_z_above_table=0.33,
            slow_xy_carry=True,
            settle_steps_place=28,
            pos_threshold_coarse=0.052,
        )
    if task in ("g3", "grid3"):
        n = 3
        h = 2
        slots = grid_stack_world_slots(env, n, n, h)
        pick = pick_override or "color"
        jobs = _jobs_from_slots(env, slots, pick)
        return GenericAssemblyPolicy(
            env,
            jobs,
            inter_cube_gap=gap,
            ascend_above_each_slot=True,
            ascend_clear_above_placed_z=0.40,
            hover_slot_z_extra=0.18,
            transit_min_z_above_table=0.35,
            slow_xy_carry=True,
            settle_steps_place=30,
            pos_threshold_coarse=0.055,
            final_park_z_offset=0.38,
        )
    if task in ("smiley", "face"):
        slots = smiley_world_slots(env)
        pick = pick_override or "color"
        jobs = _jobs_from_slots(env, slots, pick)
        return GenericAssemblyPolicy(
            env,
            jobs,
            ascend_above_each_slot=False,
            clear_height_above_table=0.24,
            hover_slot_z_extra=0.08,
            transit_min_z_above_table=0.26,
            slow_xy_carry=True,
            settle_steps_place=26,
            pos_threshold_coarse=0.05,
        )
    return None


def _cube_count_for_task(task: str) -> int:
    if task in ("h", "row", "horizontal", "hc", "hcolor", "rowc", "sort", "colorsort"):
        return 5
    if task in ("v", "stack", "vertical"):
        return 3
    if task in ("g2", "grid2"):
        return grid_cube_count(2, 2, 2)
    if task in ("g3", "grid3"):
        return grid_cube_count(3, 3, 2)
    if task in ("smiley", "face"):
        return smiley_cube_count()
    raise KeyError(task)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-render",
        action="store_true",
        help="No on-screen viewer (faster; suitable for headless runs).",
    )
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--hold-seconds", type=float, default=4.0)
    ap.add_argument(
        "--pick",
        choices=("color", "x"),
        default=None,
        help="Override pick order for h / v / grids / smiley (default: color for v/g/smiley, x for h).",
    )
    ap.add_argument(
        "--one-shot",
        type=str,
        default=None,
        help="Non-interactive: e.g. 'blue v' then exit.",
    )
    args = ap.parse_args()
    render = not args.no_render

    def run_one(task: str, color_name: str):
        rgba = cycle_for_preset(color_name)
        n_cubes = _cube_count_for_task(task)
        profile = (
            "table_uniform"
            if n_cubes > 14
            else "calibration"
        )
        env = _make_env(
            cube_count=n_cubes,
            rgba_cycle=rgba,
            horizon=args.max_steps + 50,
            render=render,
            placement_profile=profile,
        )
        pick_ov = args.pick
        policy = _build_policy(task, env, pick_ov)
        if policy is None:
            print("Unknown task:", task)
            env.close()
            return
        print(f"  Task={task!r} color={color_name!r} cubes={n_cubes}")
        _run_policy(env, policy, max_steps=args.max_steps, render=render, hold_seconds=args.hold_seconds)
        env.close()

    if args.one_shot:
        task, color = _parse_line(args.one_shot)
        if task is None:
            print("Could not parse task from:", args.one_shot)
            sys.exit(1)
        run_one(task, color)
        return

    print(__doc__)
    while True:
        try:
            line = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("q", "quit", "exit"):
            break
        if line.lower() in ("help", "?"):
            print(
                "  Tasks: h hc|sort v smiley g2 g3  |  Colors: default blue red green yellow magenta"
            )
            continue
        task, color = _parse_line(line)
        if task is None:
            print("  Unrecognized; type help for a short list.")
            continue
        try:
            run_one(task, color)
        except Exception as e:
            print("  Error:", e)


if __name__ == "__main__":
    main()
