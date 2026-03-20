#!/usr/bin/env python3
"""Row checker math, spawn=horizontal_line, then hack poses and see verify flip."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import robosuite  # noqa: E402

import line_assembly  # noqa: E402, F401
from line_assembly.layout_verify import check_horizontal_adjacent_line  # noqa: E402
from line_assembly.horizontal_line_policy import pick_indices_in_palette_order  # noqa: E402


def _test_pick_order_by_color():
    # 6 cubes, palette 5: indices 0,5 share palette 0; order by palette then x
    pos_x = np.array([0.5, 0.1, 0.2, 0.3, 0.4, 0.05])
    order = pick_indices_in_palette_order(6, pos_x, palette_size=5)
    # palette 0: cubes 0 (x=0.5) and 5 (x=0.05) → by x: 5, 0
    # palette 1: cube 1 x=0.1
    # palette 2: cube 2
    # palette 3: cube 3
    # palette 4: cube 4
    assert list(order) == [5, 0, 1, 2, 3, 4]
    print("pick order by color: OK")


def _test_pure_geometry():
    hx = 0.02
    h = (hx, hx, hx)
    # Perfect row along x
    pos = np.array([[i * 2 * hx, 0.0, 0.79] for i in range(4)])
    r = check_horizontal_adjacent_line(pos, h)
    assert r.ok, r.messages

    # Break y alignment
    pos_bad = pos.copy()
    pos_bad[2, 1] += 0.1
    r2 = check_horizontal_adjacent_line(pos_bad, h)
    assert not r2.ok, "expected y_span failure"

    # Wrong pitch (too far apart)
    pos_gap = np.array([[0.0, 0.0, 0.79], [0.2, 0.0, 0.79]])
    r3 = check_horizontal_adjacent_line(pos_gap, h)
    assert not r3.ok, "expected pitch failure"

    print("pure geometry: OK")


def _make_env(placement_profile: str, cubes: int = 4, horizontal_reward_mode: str = "none"):
    return robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        horizon=50,
        control_freq=20,
        cube_count=cubes,
        placement_profile=placement_profile,
        horizontal_reward_mode=horizontal_reward_mode,
    )


def _test_spawn_horizontal_line_profile():
    env = _make_env("horizontal_line", cubes=5)
    env.reset()
    ev = env.evaluate_horizontal_line()
    assert ev["ok"], (ev, env.get_cube_world_positions())
    _obs, r, _d, info = env.step(np.zeros(env.action_spec[0].shape))
    assert info["horizontal_line_ok"] == ev["ok"]
    assert r == 0.0
    env.close()
    print("horizontal_line placement + info: OK")


def _test_sparse_reward():
    env = _make_env("horizontal_line", cubes=3, horizontal_reward_mode="sparse")
    env.reset()
    _obs, r, _d, _info = env.step(np.zeros(env.action_spec[0].shape))
    assert r == 1.0
    env.close()
    print("sparse horizontal reward: OK")


def _test_manual_snap_from_random():
    env = _make_env("calibration", cubes=4)
    env.reset()

    z = env._cube_center_z_on_table()
    y = 0.02
    pitch = 2.0 * float(env.cube_half_extents[0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = -0.04
    for i in range(env.cube_count):
        env.set_cube_free_joint(i, [x0 + i * pitch, y, z], quat)
    env.sim.forward()
    ev = env.evaluate_horizontal_line()
    assert ev["ok"], (ev, env.get_cube_world_positions())
    env.close()
    print("manual snap to line: OK")


def main():
    _test_pick_order_by_color()
    _test_pure_geometry()
    _test_spawn_horizontal_line_profile()
    _test_sparse_reward()
    _test_manual_snap_from_random()
    print("\nAll horizontal verification tests passed.")


if __name__ == "__main__":
    main()
