#!/usr/bin/env python3
"""Sanity check each menu task finishes. --quick skips g2. Smiley runs before g2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import robosuite  # noqa: E402

import line_assembly  # noqa: E402, F401
from line_assembly.color_presets import cycle_for_preset  # noqa: E402

import play_task_menu as ptm  # noqa: E402


def _run(task: str, *, max_steps: int, seed: int) -> tuple[int, bool]:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except ImportError:
        pass

    rgba = cycle_for_preset("default")
    n = ptm._cube_count_for_task(task)
    profile = "table_uniform" if n > 14 else "calibration"
    env = ptm._make_env(
        cube_count=n,
        rgba_cycle=rgba,
        horizon=max_steps + 100,
        render=False,
        placement_profile=profile,
    )
    obs = env.reset()
    policy = ptm._build_policy(task, env, None)
    if policy is None:
        env.close()
        raise ValueError(f"unknown task {task!r}")
    steps = 0
    while steps < max_steps and not policy.finished:
        obs, _, done, _ = env.step(policy.get_action(obs))
        steps += 1
        if done:
            break
    ok = policy.finished
    env.close()
    return steps, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip 2×2 grid only (smiley still runs).",
    )
    ap.add_argument("--max-steps", type=int, default=90000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tasks = ["h", "hc", "v", "smiley"]
    if not args.quick:
        tasks.extend(["g2"])

    failed: list[str] = []
    for t in tasks:
        steps, ok = _run(t, max_steps=args.max_steps, seed=args.seed)
        status = "OK" if ok else "FAIL"
        print(f"  {t:8s}  steps={steps:6d}  {status}")
        if not ok:
            failed.append(t)

    if failed:
        raise SystemExit(f"formations failed: {failed}")
    print("\nAll formation tests passed.")


if __name__ == "__main__":
    main()
